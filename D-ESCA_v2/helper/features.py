from os.path import join
from helper.parser import arg_parser 
from core.Preprocessing import Feature_extractor
from config import update_config, get_cfg_defaults

if __name__ == '__main__':

    # update config based on default.yaml file
    cfg = get_cfg_defaults()
    config_file = arg_parser('Feature extracting module')
    cfg = update_config(cfg, config_file)

    # initiate a feature_extractor with all parameters from cfg
    feat_extr = Feature_extractor(
        type=cfg.PREPROCESS.TYPE, segment_len=cfg.PREPROCESS.SEGMENT_LEN, \
        audio_len=cfg.PREPROCESS.AUDIO_LEN, sample_per_file=cfg.PREPROCESS.SAMPLE_PER_FILE, \
        window_time=cfg.PREPROCESS.GAMMA.WINDOW_TIME, hop_time=cfg.PREPROCESS.GAMMA.HOP_TIME, \
        channels=cfg.PREPROCESS.GAMMA.CHANNELS, f_min=cfg.PREPROCESS.GAMMA.F_MIN, \
        sr=cfg.PREPROCESS.MEL.SR, nfft=cfg.PREPROCESS.MEL.NFFT, n_mel_band=cfg.PREPROCESS.MEL.N_BANDS
    )

    for audio_src_dir, feat_dst_dir in zip(cfg.PREPROCESS.SRC, cfg.PREPROCESS.DST):
        feat_extr.src = audio_src_dir
        feat_extr.dst = feat_dst_dir
        feat_extr.extract_feature(cfg.PREPROCESS.TYPE)
    feat_extr.save_id()