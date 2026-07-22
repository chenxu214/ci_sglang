#!/usr/bin/env bash
#
# scp_retry.sh — 基于 rsync 的断点续传 + 失败重试拷贝脚本
#
# 用法:
#   ./scp_retry.sh <本地源> <远程目标> [最大重试次数]
#
# 示例:
#   ./scp_retry.sh /data/logs/  user@host:/data/logs/
#   ./scp_retry.sh /data/big.tar user@host:/data/ 5
#
# 说明:
#   - 用 rsync 替代 scp, 原生支持断点续传:
#       * --partial        : 保留传输中断的部分文件, 下次继续而非重传
#       * --append-verify  : 续传前先校验远端已有部分与本地前缀是否一致, 防止脏追加
#       * --inplace        : 原地写入, 配合 --partial 才能真正续传
#       * --size-only      : 跳过大小已一致的文件(快速判断, 不校验内容)
#   - 每轮 rsync 失败后, 指数退避重试, 最多 MAX_RETRY 次
#   - 每轮结束后对每个文件做大小校验, 残缺的会在下一轮继续补传
#   - 跳过点开头的隐藏目录(如 .git/.ssh/.cache)
#   - 需要 ssh 免密登录或 ssh-agent

set -u

# ---------- 参数解析 ----------
if [ $# -lt 2 ]; then
    echo "用法: $0 <本地源> <远程目标> [最大重试次数]" >&2
    exit 2
fi

SRC="$1"
DST="$2"
MAX_RETRY="${3:-10}"          # 默认重试 10 次
SLEEP_BASE=2                   # 退避起始秒数
SSH_OPTS=(-o ConnectTimeout=15 -o ServerAliveInterval=5 -o ServerAliveCountMax=3)

# rsync 通过 ssh 传输, 并启用断点续传相关选项
RSYNC_OPTS=(
    -azP                        # -a 归档 -z 压缩 -P = --partial + --progress
    --inplace                   # 原地写入, 才能续传部分文件
    --append-verify             # 续传前校验远端已有前缀, 防脏追加
    --size-only                 # 大小一致即跳过(快速判断)
    --exclude='.*/'             # 跳过点开头的隐藏目录及其内容
    -e "ssh ${SSH_OPTS[*]}"
)

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

# ---------- 主重试循环 ----------
attempt=0
while [ "$attempt" -lt "$MAX_RETRY" ]; do
    attempt=$((attempt + 1))
    log "===== 第 $attempt/$MAX_RETRY 次尝试 ====="
    log "rsync $SRC -> $DST"

    # rsync 单次执行即处理整个源(目录或文件)
    # --partial + --inplace + --append-verify 保证中断后下次能续传
    if rsync "${RSYNC_OPTS[@]}" "$SRC" "$DST"; then
        # rsync 返回成功, 但仍逐文件校验大小, 防止极少数截断被误判
        log "rsync 完成, 开始大小校验..."
        bad=0

        if [ -d "$SRC" ]; then
            src_dir="${SRC%/}/"
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
            done < <(find "$src_dir" -type f -not -path '*/.*' -print0)
        else
            lf="$SRC"
            rf="${RPATH%/}/$(basename "$SRC")"
            # 若目标本身是目录路径, basename 不适用; 但 rsync 已保证布局, 这里仅尽力校验
            lsize=$(stat -c %s "$lf" 2>/dev/null || stat -f %z "$lf" 2>/dev/null)
            rsize=$(remote_size "$rf")
            if [ -z "$rsize" ] || [ "$rsize" != "$lsize" ]; then
                fail "校验失败: $lf (本地=${lsize:-?} 远端=${rsize:-空})"
                bad=1
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

    # 退避后重试
    sleep_sec=$(( SLEEP_BASE * attempt ))
    [ "$sleep_sec" -gt 60 ] && sleep_sec=60
    log "本轮未完成, ${sleep_sec}s 后重试..."
    sleep "$sleep_sec"
done

fail "已达到最大重试次数 $MAX_RETRY, 仍有文件未拷贝完成"
exit 1
