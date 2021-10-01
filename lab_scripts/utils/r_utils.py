from rpy2.robjects import r


def activate_R_envinronment(lib_path=None):
    if not lib_path:
        lib_path = "~/R/nips"
    paths = r[".libPaths"]()
    paths[0] = lib_path
    r[".libPaths"](paths)
