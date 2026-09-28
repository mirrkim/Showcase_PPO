"""One-click runtime dependency installer for the showcase app."""

import subprocess
import sys


RUNTIME_PACKAGES = [
    'numpy',
    'scipy',
    'pandas',
    'openpyxl',
    'pygame',
    'tensorflow',
]


def run(command):
    print('\n>', ' '.join(command))
    subprocess.check_call(command)


def main():
    python = sys.executable
    print(f'Python: {python}')
    print('Installing showcase runtime dependencies...')

    run([python, '-m', 'pip', 'install', *RUNTIME_PACKAGES])

    # 기존 PyTorch 패키지를 정리한 뒤
    # RTX 5070용 CUDA 12.8 빌드를 새로 설치합니다.
    run([
        python, '-m', 'pip', 'uninstall', '-y',
        'torch', 'torchvision', 'torchaudio',
    ])
    run([
        python, '-m', 'pip', 'install',
        'torch', 'torchvision', 'torchaudio',
        '--index-url', 'https://download.pytorch.org/whl/cu128',
    ])

    print('\nInstallation complete.')
    print('Run run_showcase.py to start the application.')


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f'\nInstallation failed with exit code {error.returncode}.')
        raise
    finally:
        if sys.stdin.isatty():
            input('\nPress Enter to close...')
