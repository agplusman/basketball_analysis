import os
import argparse
from configs import STUBS_DEFAULT_PATH, OUTPUT_VIDEO_PATH
from pipeline_runner import run_analysis

def parse_args():
    parser = argparse.ArgumentParser(description='Basketball Video Analysis')
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('--output_video', type=str, default=OUTPUT_VIDEO_PATH, 
                        help='Path to output video file')
    parser.add_argument('--stub_path', type=str, default=STUBS_DEFAULT_PATH,
                        help='Path to stub directory')
    return parser.parse_args()

def main():
    args = parse_args()
    run_analysis(
        input_video_path=args.input_video,
        output_video_path=args.output_video,
        stub_path=args.stub_path,
        use_stubs=True,
    )

if __name__ == '__main__':
    main()
    
