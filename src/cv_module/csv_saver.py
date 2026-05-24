import pandas as pd
import os

class CSVSaver:
    def __init__(self, output_dir='outputs/csv/'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_frame_data(self, frame_results, video_name):
        # 展开 behavior_counts 字典
        expanded_results = []
        for result in frame_results:
            row = {
                'timestamp': result['timestamp'],
                'frame_num': result['frame_num'],
                'total_stu': result['total_stu'],
                'is_valid': result.get('is_valid', True)
            }
            # 展开行为计数
            behavior_counts = result.get('behavior_counts', {})
            for behavior in ['focus_listen', 'study_bow', 'empty_mind', 'sleep_stu',
                             'look_side', 'talk_discuss', 'talk_private', 'stand_up',
                             'loose_stu', 'phone_game']:
                row[behavior] = behavior_counts.get(behavior, 0)
            expanded_results.append(row)

        df = pd.DataFrame(expanded_results)

        columns = [
            'timestamp', 'frame_num', 'total_stu',
            'focus_listen', 'study_bow', 'empty_mind', 'sleep_stu',
            'look_side', 'talk_discuss', 'talk_private',
            'stand_up', 'loose_stu', 'phone_game',
            'is_valid'
        ]

        available_cols = [col for col in columns if col in df.columns]
        df = df[available_cols]

        output_path = os.path.join(self.output_dir, f"{video_name}_raw.csv")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Data saved: {output_path} ({len(df)} frames)")

        return output_path

    def load_frame_data(self, video_name):
        output_path = os.path.join(self.output_dir, f"{video_name}_raw.csv")
        if os.path.exists(output_path):
            return pd.read_csv(output_path, encoding='utf-8-sig')
        return None
