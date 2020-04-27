from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    json_path = '/home/jithin/datasets/imageCaptioning/captions/dataset_flickr8k.json'
    img_folder = '/home/jithin/datasets/imageCaptioning/flicker8k/Flicker8k_Dataset'
    output_folder = './pre_processed'
    create_input_files(dataset='flickr8k',
                       karpathy_json_path=json_path,
                       image_folder=img_folder,
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder=output_folder,
                       max_len=50)
