import os
from PIL import Image
import csv
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def process_images_in_directory(processor, input_dir, output_dir, log_file='testing_results/logs/processing_log.csv'):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_files = [f for f in os.listdir(input_dir)]
    total_images = len(image_files)  
    
    with open(log_file, 'w', newline='') as csvfile:
        fieldnames = ['Image Name', 'Resolution', 'Time Total', 'Time Shader Init',
                      'Time Shader Processing', 'Time Loading Image', 'Time Saving Image']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        with tqdm(total=total_images, desc="Processing Images") as pbar:
            for filename in image_files:
                input_image_path = os.path.join(input_dir, filename)
                
                output_image_name = filename
                output_image_path = os.path.join(output_dir, output_image_name)

                try:
                    with Image.open(input_image_path) as img:
                        width, height = img.size
                        resolution = f"{width}x{height}"

                    perf_log = processor.process(input_image_path, output_image_path)

                    log_entry = {
                        'Image Name': filename,
                        'Resolution': resolution,
                        'Time Total': perf_log.get('time_total', 0),
                        'Time Shader Init': perf_log.get('time_shader_init', 0),
                        'Time Shader Processing': perf_log.get('time_shader_processing', 0),
                        'Time Loading Image': perf_log.get('time_loading_image', 0),
                        'Time Saving Image': perf_log.get('time_saving_image', 0)
                    }

                    writer.writerow(log_entry)
                    pbar.update(1)
                    pbar.set_description(f"Processing {filename}")

                except Exception as e:
                    print(f"Error processing image {filename}: {e}")
                    return

def plot_time(log_file, 
              output_file='testing_results/plots/time_plot.png',
              plot_title='Среднее время обработки',
              exclude_time_columns=(),
              show_column_numbers=()
            ):

    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"Error: File '{log_file}' not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: File '{log_file}' is empty.")
        return
    except Exception as e:
        print(f"Error reading file '{log_file}': {e}")
        return

    grouped_data = df.groupby('Resolution')[
        ['Time Total', 'Time Shader Init', 'Time Loading Image', 'Time Saving Image', 'Time Shader Processing']
    ].mean()

    # Sort by width
    try:
        grouped_data = grouped_data.sort_index(key=lambda x: [int(s.split('x')[0]) for s in x], ascending=False) 
    except ValueError:
        print('Error, invalid resolution format. Expected format: WxH.')
        return

    # Plot
    base_time_columns = ['Time Total', 'Time Shader Init', 'Time Loading Image', 'Time Saving Image', 'Time Shader Processing',]
    time_columns = [column for column in base_time_columns if not column in exclude_time_columns]
    
    fig, ax = plt.subplots(figsize=(12, 6))

    bar_width = 0.8 / len(time_columns)
    positions = np.arange(len(grouped_data.index))

    for i, column in enumerate(time_columns):
        bars = ax.bar(positions + i * bar_width, grouped_data[column], width=bar_width, label=column)

        if column in show_column_numbers:
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2),
                        ha='center', va='bottom')

    ax.set_xlabel('Разрешение изображения')
    ax.set_ylabel('Среднее время (ms)')
    ax.set_title(plot_title)
    ax.set_xticks(positions + (len(time_columns) - 1) * bar_width / 2)
    ax.set_xticklabels(grouped_data.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--')
    plt.tight_layout()

    plt.savefig(output_file)
    print(f'Plot saved in {output_file}')