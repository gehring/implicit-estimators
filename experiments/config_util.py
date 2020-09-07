import gin.config


def gin_operative_dict():
    registry = gin.config._REGISTRY
    config = {}
    for (scope, selector), config_entry in gin.config._OPERATIVE_CONFIG.items():
        configurable_ = registry[selector]
        fn = configurable_.fn_or_cls
        if fn == gin.config.macro:
            config[scope] = config_entry["value"]
        elif fn != gin.config._retrieve_constant:
            minimal_selector = registry.minimal_selector(configurable_.selector)
            scoped_selector = (scope + '/' if scope else '') + minimal_selector
            for k, v in config_entry.items():
                key = f"{scoped_selector}.{k}"
                config[key] = gin.config._format_value(v)
    return config
