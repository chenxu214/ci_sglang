#!/usr/bin/env bash
#
# scp_retry.sh — 基于 rsync 的断点续传 + 失败重试拷贝脚本
#
# 用法:
#   ./scp_retry.sh <本地源> <远程目标> [最大重试次数] [文件名前缀列表]
#
# 示例:
#   ./scp_retry.sh /data/logs/  user@host:/data/logs/
#   ./scp_retry.sh /data/big.tar user@host:/data/ 5
#   ./scp_retry.sh /data/logs/  user@host:/data/logs/ 10 "log-,err-,access_"
#
# 说明:
#   - 用 rsync 替代 scp, 原生支持断点续传:
#       * --partial        : 保留传输中断的部分文件, 下次继续而非重传
#       * --append-verify  : 续传前先校验远端已有部分与本地前缀是否一致, 防止脏追加
#       * --inplace        : 原地写入, 配合 --partial 才能真正续传
#       * --checksum       : 对所有文件算 MD5, 大小一致但内容不同也会重传, 保证内容一致性
#   - 每轮 rsync 失败后, 固定等待 3s 重试, 最多 MAX_RETRY 次
#   - 每轮结束后对每个文件做大小校验, 残缺的会在下一轮继续补传
#   - 跳过点开头的隐藏目录(如 .git/.ssh/.cache)
#   - 可选第 4 个参数: 文件名前缀列表(逗号分隔), 仅拷贝文件名匹配任一前缀的文件;
#     不传则拷贝全部
#   - 需要 ssh 免密登录或 ssh-agent

set -u

# ---------- 参数解析 ----------
if [ $# -lt 2 ]; then
    echo "用法: $0 <本地源> <远程目标> [最大重试次数] [文件名前缀列表]" >&2
    exit 2
fi

SRC="$1"
DST="$2"
MAX_RETRY="${3:-1000}"          # 默认重试 1000 次
PREFIXES="${4:-}"               # 可选: 文件名前缀列表, 逗号分隔, 如 "log-,err-"
SSH_OPTS=(-o ConnectTimeout=15 -o ServerAliveInterval=5 -o ServerAliveCountMax=3)

# rsync 通过 ssh 传输, 并启用断点续传相关选项
RSYNC_OPTS=(
    -azP                        # -a 归档 -z 压缩 -P = --partial + --progress
    --inplace                   # 原地写入, 才能续传部分文件
    --append-verify             # 续传前校验远端已有前缀, 防脏追加
    --checksum                  # 校验 MD5, 大小一致但内容不同也会重传, 保证内容一致性
    --exclude='.*/'             # 跳过点开头的隐藏目录及其内容
    -e "ssh ${SSH_OPTS[*]}"
)

# 若指定了文件名前缀, 构造 rsync 过滤规则: 先放行目录以便递归, 再按前缀 include, 最后排除其余
# rsync 过滤为 first-match-wins, 顺序很关键
if [ -n "$PREFIXES" ]; then
    RSYNC_OPTS+=(--include='*/')        # 允许递归进入子目录
    IFS=',' read -ra _prefs <<< "$PREFIXES"
    for p in "${_prefs[@]}"; do
        # 跳过空前缀(连续逗号或末尾逗号)
        [ -n "$p" ] && RSYNC_OPTS+=(--include="${p}*")
    done
    RSYNC_OPTS+=(--exclude='*')         # 排除所有未匹配前缀的文件
fi

log()  { echo "[$(date '+%F %T')] $*"; }
fail() { log "错误: $*" >&2; }

# ---------- 前置检查 ----------
command -v rsync >/dev/null 2>&1 || { fail "未找到 rsync, 请先安装"; exit 1; }

# 仅支持"本地 -> 远程"方向
is_remote() { [[ "$1" == *:* ]]; }
if is_remote "$SRC"; then
    fail "本脚本仅支持 本地->远程 拷贝; 源不能是远端路径"
    exit 2
fi
if ! is_remote "$DST"; then
    fail "目标必须是远端路径, 形如 user@host:/path"
    exit 2
fi

# ---------- 拆分远端路径(用于大小校验) ----------
RHOST="${DST%%:*}"
RPATH="${DST#*:}"

remote_exec() {
    ssh "${SSH_OPTS[@]}" "$RHOST" "$@"
}

# 取远端文件大小(字节), 不存在或失败返回空
remote_size() {
    local rf="$1"
    remote_exec "stat -c %s '$rf' 2>/dev/null || stat -f %z '$rf' 2>/dev/null"
}

# 构造 find 的文件名过滤参数(与 rsync include 规则保持一致)
# 有前缀: \( -name 'p1*' -o -name 'p2*' \); 无前缀: 空数组
FIND_NAME=()
if [ -n "$PREFIXES" ]; then
    _first=1
    IFS=',' read -ra _prefs <<< "$PREFIXES"
    for p in "${_prefs[@]}"; do
        [ -z "$p" ] && continue
        if [ "$_first" -eq 1 ]; then
            FIND_NAME+=(-name "${p}*")
            _first=0
        else
            FIND_NAME+=(-o -name "${p}*")
        fi
    done
fi

# 检查文件名是否匹配任一前缀; 未指定前缀时总是匹配
# 返回 0=匹配 1=不匹配
match_prefix() {
    local fname="$1"
    [ -z "$PREFIXES" ] && return 0
    local p
    IFS=',' read -ra _prefs <<< "$PREFIXES"
    for p in "${_prefs[@]}"; do
        [ -z "$p" ] && continue
        [[ "$fname" == "${p}"* ]] && return 0
    done
    return 1
}

# ---------- 主重试循环 ----------
attempt=0
while [ "$attempt" -lt "$MAX_RETRY" ]; do
    attempt=$((attempt + 1))
    log "===== 第 $attempt/$MAX_RETRY 次尝试 ====="
    if [ -n "$PREFIXES" ]; then
        log "rsync $SRC -> $DST (前缀过滤: $PREFIXES)"
    else
        log "rsync $SRC -> $DST"
    fi

    # rsync 单次执行即处理整个源(目录或文件)
    # --partial + --inplace + --append-verify 保证中断后下次能续传
    if rsync "${RSYNC_OPTS[@]}" "$SRC" "$DST"; then
        # rsync 返回成功, 但仍逐文件校验大小, 防止极少数截断被误判
        log "rsync 完成, 开始大小校验..."
        bad=0

        if [ -d "$SRC" ]; then
            src_dir="${SRC%/}/"
            # 有前缀过滤时, find 只列出匹配任一前缀的文件; 无前缀时列出全部
            if [ ${#FIND_NAME[@]} -gt 0 ]; then
                _find_cmd=(find "$src_dir" -type f \( "${FIND_NAME[@]}" \) -not -path '*/.*' -print0)
            else
                _find_cmd=(find "$src_dir" -type f -not -path '*/.*' -print0)
            fi
            while IFS= read -r -d '' f; do
                rel="${f#$src_dir}"
                lf="$f"
                rf="${RPATH%/}/$rel"
                lsize=$(stat -c %s "$lf" 2>/dev/null || stat -f %z "$lf" 2>/dev/null)
                rsize=$(remote_size "$rf")
                if [ -z "$rsize" ] || [ "$rsize" != "$lsize" ]; then
                    fail "校验失败: $rel (本地=${lsize:-?} 远端=${rsize:-空})"
                    bad=1
                fi
            done < <("${_find_cmd[@]}")
        else
            lf="$SRC"
            fname=$(basename "$SRC")
            if ! match_prefix "$fname"; then
                log "文件 $fname 不匹配前缀, 跳过"
            else
                rf="${RPATH%/}/$(basename "$SRC")"
                # 若目标本身是目录路径, basename 不适用; 但 rsync 已保证布局, 这里仅尽力校验
                lsize=$(stat -c %s "$lf" 2>/dev/null || stat -f %z "$lf" 2>/dev/null)
                rsize=$(remote_size "$rf")
                if [ -z "$rsize" ] || [ "$rsize" != "$lsize" ]; then
                    fail "校验失败: $lf (本地=${lsize:-?} 远端=${rsize:-空})"
                    bad=1
                fi
            fi
        fi

        if [ "$bad" -eq 0 ]; then
            log "全部文件大小一致, 拷贝完成"
            exit 0
        fi
        log "部分文件校验失败, 将进入下一轮重试补传"
    else
        fail "rsync 退出码 $?, 本轮失败"
    fi

    # 固定等待 3s 后重试
    log "本轮未完成, 3s 后重试..."
    sleep 3
done

fail "已达到最大重试次数 $MAX_RETRY, 仍有文件未拷贝完成"
exit 1
