#!/usr/bin/env bash
set -euo pipefail
clip_id="$1"; source_y4m="$2"; fps="$3"; config="$4"; crf="$5"; frames="$6"; work="$7"; points="$8"
mkdir -p "$work/$clip_id"
out="$work/$clip_id/${config}.crf${crf}.mkv"
time_file="$work/$clip_id/${config}.crf${crf}.time"
encode_log="$work/$clip_id/${config}.crf${crf}.encode.log"
rc_file="$work/$clip_id/${config}.crf${crf}.encode.rc"
case "$config" in
  baseline_fast) profile=(-cpu-used 8 -aq-mode 1 -tune psnr) ;;
  complexity_aq) profile=(-cpu-used 8 -aq-mode 2 -tune psnr) ;;
  quality_slow) profile=(-cpu-used 6 -aq-mode 1 -tune psnr) ;;
  screen_tools) profile=(-cpu-used 8 -aq-mode 1 -tune psnr -enable-intrabc 1 -enable-palette 1) ;;
  *) echo "unknown fixed profile: $config" >&2; exit 2 ;;
esac
set +e
/usr/bin/time -f '%e' -o "$time_file" ffmpeg -nostdin -hide_banner -loglevel verbose -y -i "$source_y4m" -map 0:v:0 -frames:v "$frames" -an -c:v libaom-av1 -usage good -crf "$crf" -b:v 0 -row-mt 1 -threads 2 -g "$frames" "${profile[@]}" -pix_fmt yuv420p "$out" 2> "$encode_log"
rc=$?
set -e
printf '%s\n' "$rc" > "$rc_file"
if (( rc != 0 )); then
  exit "$rc"
fi
seconds=$(cat "$time_file")
bytes=$(stat -c%s "$out")
bitrate=$(python -c 'import sys; print(float(sys.argv[1])*8.0/(float(sys.argv[2])/float(sys.argv[3]))/1000.0)' "$bytes" "$frames" "$fps")
ffmpeg -nostdin -hide_banner -i "$out" -i "$source_y4m" -lavfi '[0:v]setpts=PTS-STARTPTS[d];[1:v]setpts=PTS-STARTPTS[r];[d][r]psnr' -frames:v "$frames" -f null - 2> "$out.psnr.log"
ffmpeg -nostdin -hide_banner -i "$out" -i "$source_y4m" -lavfi '[0:v]setpts=PTS-STARTPTS[d];[1:v]setpts=PTS-STARTPTS[r];[d][r]ssim' -frames:v "$frames" -f null - 2> "$out.ssim.log"
psnr=$(grep -oE 'PSNR y:[0-9.+-]+' "$out.psnr.log" | tail -1 | cut -d: -f2)
ssim=$(grep -oE 'SSIM Y:[0-9.+-]+' "$out.ssim.log" | tail -1 | cut -d: -f2)
decoded_frames=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nw=1:nk=1 "$out")
ffmpeg -nostdin -hide_banner -loglevel error -i "$out" -map 0:v:0 -frames:v "$frames" -f framehash -hash sha256 "$out.framehash"
bitstream_sha=$(sha256sum "$out" | cut -d' ' -f1)
decoded_sha=$(sha256sum "$out.framehash" | cut -d' ' -f1)
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$clip_id" "$config" "$crf" "$bytes" "$bitrate" "$seconds" "$psnr" "$ssim" "$decoded_frames" "$bitstream_sha" "$decoded_sha" >> "$points"
