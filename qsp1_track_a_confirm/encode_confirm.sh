#!/usr/bin/env bash
set -euo pipefail
clip="$1"; split="$2"; source_y4m="$3"; fps="$4"; config="$5"; crf="$6"; frames="$7"; trial="$8"; work="$9"; points="${10}"; hash_tool="${11}"
mkdir -p "$work/$clip"
out="$work/$clip/${config}.crf${crf}.trial${trial}.ivf"
time_file="$out.time"; enc_log="$out.encode.log"
case "$config" in
  baseline_fast) profile=(-cpu-used 8 -aq-mode 1 -tune psnr) ;;
  complexity_aq) profile=(-cpu-used 8 -aq-mode 2 -tune psnr) ;;
  quality_slow) profile=(-cpu-used 6 -aq-mode 1 -tune psnr) ;;
  screen_tools) profile=(-cpu-used 8 -aq-mode 1 -tune psnr -enable-intrabc 1 -enable-palette 1) ;;
  *) echo "unknown profile: $config" >&2; exit 2 ;;
esac
set +e
/usr/bin/time -f '%e' -o "$time_file" ffmpeg -nostdin -hide_banner -loglevel warning -y -i "$source_y4m" -map 0:v:0 -frames:v "$frames" -an -c:v libaom-av1 -usage good -crf "$crf" -b:v 0 -row-mt 1 -threads 2 -g "$frames" "${profile[@]}" -pix_fmt yuv420p -color_range tv -colorspace smpte170m -color_primaries smpte170m -color_trc smpte170m -f ivf "$out" >"$enc_log" 2>&1
rc=$?
set -e
if [[ $rc -ne 0 ]]; then echo "encode failed $clip $config $crf trial $trial rc=$rc" >&2; cat "$enc_log" >&2; exit "$rc"; fi
seconds=$(cat "$time_file"); bytes=$(stat -c%s "$out")
hash_json=$(python "$hash_tool" "$out")
payload_sha=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["payload_sha256"])' "$hash_json")
payload_bytes=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["payload_bytes"])' "$hash_json")
payload_frames=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["frames"])' "$hash_json")
bitrate=$(python -c 'import sys; print(float(sys.argv[1])*8.0/(float(sys.argv[2])/float(sys.argv[3]))/1000.0)' "$bytes" "$frames" "$fps")
framehash="$out.framehash"
ffmpeg -nostdin -hide_banner -loglevel error -i "$out" -map 0:v:0 -frames:v "$frames" -f framehash -hash sha256 "$framehash"
decoded_sha=$(sha256sum "$framehash" | cut -d' ' -f1)
decoded_frames=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nw=1:nk=1 "$out")
psnr=''; ssim=''; warning_count=0
if [[ "$trial" == "1" ]]; then
  psnr_log="$out.psnr.log"; ssim_log="$out.ssim.log"
  norm_d="settb=AVTB,setpts=N/(${fps}*TB),format=yuv420p,setrange=limited"
  norm_r="settb=AVTB,setpts=N/(${fps}*TB),format=yuv420p,setrange=limited"
  ffmpeg -nostdin -hide_banner -i "$out" -i "$source_y4m" -filter_complex "[0:v]${norm_d}[d];[1:v]${norm_r}[r];[d][r]psnr" -frames:v "$frames" -f null - 2>"$psnr_log"
  ffmpeg -nostdin -hide_banner -i "$out" -i "$source_y4m" -filter_complex "[0:v]${norm_d}[d];[1:v]${norm_r}[r];[d][r]ssim" -frames:v "$frames" -f null - 2>"$ssim_log"
  psnr=$(grep -oE 'PSNR y:[0-9.+-]+' "$psnr_log" | tail -1 | cut -d: -f2)
  ssim=$(grep -oE 'SSIM Y:[0-9.+-]+' "$ssim_log" | tail -1 | cut -d: -f2)
  warning_count=$(cat "$psnr_log" "$ssim_log" | grep -Eic 'warning|mismatch|deprecated|range' || true)
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$clip" "$split" "$config" "$crf" "$trial" "$bytes" "$payload_bytes" "$bitrate" "$seconds" "$psnr" "$ssim" "$decoded_frames" "$payload_frames" "$payload_sha" "$decoded_sha" "$warning_count" >> "$points"
if [[ "$trial" != "1" ]]; then rm -f "$out" "$framehash" "$enc_log" "$time_file"; fi
