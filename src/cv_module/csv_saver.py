import pandas as pd
import os

class CSVSaver:
    def __init__(self, output_dir='cache_csv/'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_frame_data(self, frame_results, video_name):
        df = pd.DataFrame(frame_results)

        columns = [
            'timestamp', 'frame_num', 'is_valid', 'invalid_reason', 'total_stu',
            'focus_listen', 'study_bow', 'empty_mind', 'sleep_stu',
            'look_side', 'talk_discuss', 'talk_private',
            'stand_up', 'loose_stu', 'phone_game'
        ]

        available_cols = [col for col in columns if col in df.columns]
        df = df[available_cols]

        output_path = f"{self.output_dir}/{video_name}_raw.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Data saved: {output_path} ({len(df)} frames)")

        return output_path

    def load_frame_data(self, video_name):
        output_path = f"{self.output_dir}/{video_name}_raw.csv"
        if os.path.exists(output_path):
            return pd.read_csv(output_path, encoding='utf-8-sig')
        return None
