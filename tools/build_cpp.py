# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Neil Vaytet

import os
import argparse
import shutil
import subprocess
import multiprocessing
import sys

parser = argparse.ArgumentParser(description='Build C++ library and run tests')
parser.add_argument('--prefix', default='install')
parser.add_argument('--source_dir', default='.')
parser.add_argument('--build_dir', default='build')


def run_command(cmd, shell):
    """
    Run a command (supplied as a list) using subprocess.check_call
    """
    os.write(1, "{}\n".format(' '.join(cmd)).encode())
    return subprocess.check_call(cmd, stderr=subprocess.STDOUT, shell=shell)


def main(prefix='install', build_dir='build', source_dir='.'):
    """
    Platform-independent function to run cmake, build, install and C++ tests.
    """
    debug_build = "SCIPP_DEBUG_BUILD" in os.environ

    print("Building in {} configuration".format(
        "debug" if debug_build else "release"))

    # Get the platform name: 'linux', 'darwin' (osx), or 'win32'.
    platform = sys.platform

    # Set up absolute directory paths
    source_dir = os.path.abspath(source_dir)
    prefix = os.path.abspath(prefix)
    build_dir = os.path.abspath(build_dir)

    # Default options
    shell = False
    parallel_flag = '-j{}'.format(multiprocessing.cpu_count())
    build_config = ''

    # Some flags use a syntax with a space separator instead of '='
    use_space = ['-G', '-A']

    # Default cmake flags
    cmake_flags = {
        '-G': 'Ninja',
        '-DPYTHON_EXECUTABLE': shutil.which("python"),
        '-DCMAKE_INSTALL_PREFIX': prefix,
        '-DWITH_CTEST': 'OFF',
        '-DCMAKE_INTERPROCEDURAL_OPTIMIZATION': 'OFF' if debug_build else "ON",
        '-DCMAKE_BUILD_TYPE': 'Debug' if debug_build else "Release",
    }

    if debug_build:
        cmake_flags["-DDYNAMIC_LIB"] = "ON"

    if platform == 'darwin':
        cmake_flags.update({'-DCMAKE_INTERPROCEDURAL_OPTIMIZATION': 'OFF'})
        osxversion = os.environ.get('OSX_VERSION')
        if osxversion is not None:
            cmake_flags.update({
                '-DCMAKE_OSX_DEPLOYMENT_TARGET':
                osxversion,
                '-DCMAKE_OSX_SYSROOT':
                os.path.join('/Applications', 'Xcode.app', 'Contents',
                             'Developer', 'Platforms', 'MacOSX.platform',
                             'Developer', 'SDKs',
                             'MacOSX{}.sdk'.format(osxversion))
            })

    if platform == 'win32':
        cmake_flags.update({'-G': 'Visual Studio 16 2019', '-A': 'x64'})
        shell = True
        if debug_build:
            build_config = 'Debug'
            # TBB not available for windows-debug
            cmake_flags['-DDISABLE_MULTI_THREADING'] = "ON"
        else:
            build_config = 'Release'

    # Additional flags for --build commands
    build_flags = [parallel_flag]
    if len(build_config) > 0:
        build_flags += ['--config', build_config]

    # Parse cmake flags
    flags_list = []
    for key, value in cmake_flags.items():
        if key in use_space:
            flags_list += [key, value]
        else:
            flags_list.append('{}={}'.format(key, value))

    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
    os.chdir(build_dir)

    # Run cmake
    run_command(['cmake'] + flags_list + [source_dir], shell=shell)

    # Show cmake settings
    run_command(['cmake', '-B', '.', '-S', source_dir, '-LA'], shell=shell)

    # Compile benchmarks, C++ tests, and python library
    for target in ['all-benchmarks', 'all-tests', 'install']:
        run_command(['cmake', '--build', '.', '--target', target] +
                    build_flags,
                    shell=shell)

    # Run C++ tests
    for test in [
            'scipp-common-test', 'scipp-units-test', 'scipp-core-test',
            'scipp-variable-test', 'scipp-dataset-test'
    ]:
        run_command([os.path.join('bin', build_config, test)], shell=shell)


if __name__ == '__main__':
    args = parser.parse_args()
    main(prefix=args.prefix,
         build_dir=args.build_dir,
         source_dir=args.source_dir)
