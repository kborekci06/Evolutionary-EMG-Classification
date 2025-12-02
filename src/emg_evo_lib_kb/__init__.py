from .preprocessing import (load_emg_file, iter_emg_files, segment_gestures, estimate_sampling_rate,
                            channel_features, compute_segment_features, build_feature_dataset)
from .control_nn import (load_data_and_split, build_mlp_classifier, plot_confusion_matrix, 
                         run_control_nn)
from .neat_emg import (load_data_and_split_neat, evaluate_genome, eval_genomes, plot_fitness_stats,
                       run_neat_emg)