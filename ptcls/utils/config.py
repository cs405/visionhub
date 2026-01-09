import yaml
import copy

class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = AttrDict(value)
        self[key] = value

    def __deepcopy__(self, content):
        return AttrDict(copy.deepcopy(dict(self)))

def create_attr_dict(yaml_config):
    """Recursively convert dict to AttrDict.

    - dict -> AttrDict
    - list/tuple: recursively convert items
    - None/other scalar: return as-is
    """
    if yaml_config is None:
        return None

    if isinstance(yaml_config, (list, tuple)):
        converted = []
        for v in yaml_config:
            converted.append(create_attr_dict(v))
        return converted

    if isinstance(yaml_config, dict):
        for key, value in list(yaml_config.items()):
            yaml_config[key] = create_attr_dict(value)
        return AttrDict(yaml_config)

    return yaml_config

def parse_config(cfg_file):
    with open(cfg_file, 'rb') as f:
        config = yaml.safe_load(f)
    return config

def print_dict(d, delimiter=0):
    from . import logger
    # None 也可能出现在 yaml 中（例如 rec_inference_model_dir: null）
    if d is None:
        logger.info("{}{}".format('    ' * delimiter, "null"))
        return

    if isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            if isinstance(v, dict) or isinstance(v, (list, tuple)):
                logger.info("{}- [{}] :".format('    ' * delimiter, i))
                print_dict(v, delimiter + 1)
            else:
                logger.info("{}- {}".format('    ' * delimiter, v))
        return

    for k, v in sorted(d.items()):
        if isinstance(v, dict) or isinstance(v, (list, tuple)):
            logger.info("{}{} : ".format('    ' * delimiter, k))
            print_dict(v, delimiter + 1)
        elif v is None:
            logger.info("{}{} : null".format('    ' * delimiter, k))
        else:
            logger.info("{}{} : {}".format('    ' * delimiter, k, v))

def print_config(config):
    from . import logger
    logger.info("----------- Configuration -----------")
    print_dict(config)
    logger.info("-------------------------------------")

def override(dl, ks, v):
    def str2num(v):
        try:
            return eval(v)
        except Exception:
            return v

    for i, k in enumerate(ks):
        if k not in dl:
            dl[k] = {}
        if i == len(ks) - 1:
            dl[k] = str2num(v)
        else:
            dl = dl[k]

def override_config(config, options=None):
    """Override config values.

    Always returns the (possibly modified) config.
    """
    if options:
        for opt in options:
            assert '=' in opt, (
                "The format of personal options should be 'K=V', "
                "but your input is '{}'".format(opt))
            k, v = opt.split('=', 1)
            override(config, k.split('.'), v)
    return config


def get_config(fname, overrides=None, show=False):
    config = parse_config(fname)
    config = override_config(config, overrides)
    if show:
        print_config(config)
    config = create_attr_dict(config)
    return config


def _self_check():
    """Basic sanity checks for config helpers (manual call)."""
    cfg = {"Global": {"a": 1}, "X": None, "L": [{"A": 1}, None]}
    cfg = override_config(cfg, None)
    assert cfg is not None
    ad = create_attr_dict(cfg)
    assert ad is not None
    assert ad.Global["a"] == 1
    assert ad.X is None
    assert isinstance(ad.L, list)
