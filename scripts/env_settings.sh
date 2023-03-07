#!/bin/bash
#
# env.sh 
# 
# The script for loading and unloading a custom shell envrionment.
#
# by Kewei Zhang 2022/3/28 
# 
# To load the env:
#     $ source env_settings.sh
# To unloading the env:
#     $ env-deactivate
# 

env_path="$(pwd)/$0"
base_path="$(dirname $env_path)"

function env-activate {
    if [[ ! -z $_ENV_CURRENT ]]; then
        echo "ERROR: Already activated: $env_path"
        echo "do \`env-deactivate\` first."
        return
    fi

    echo "-- Activating shell env in $env_path --"
    echo "run \`env-deactivate\` to restore the origin env."
    echo

    export _ENV_CURRENT=$env_path
    export _ENV_OLD_PATH=$PATH
    export _ENV_OLD_LIBPATH=$LIBRARY_PATH
    export _ENV_OLD_LDLIBPATH=$LD_LIBRARY_PATH
    # add more sys parameters here
    # activation
    

}

function env-deactivate {
    echo "-- Deactivating shell env in $env_path --"
    deactivate
    unset _ENV_CURRENT
    # add more sys parameters here
    export PATH=$_ENV_OLD_PATH
    export LIBRARY_PATH=$_ENV_OLD_LIBPATH
    export LD_LIBRARY_PATH=$_ENV_OLD_LDLIBPATH
    unset _ENV_OLD_PATH
    unset _ENV_OLD_LIBPATH
    unset _ENV_OLD_LDLIBPATH
    echo "-- Env is Deactivated from $env_path --"
}

function deactivate {
    # deactivate the specified parameters.
}

env-activate
