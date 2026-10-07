# -*- mode: python ; coding: utf-8 -*-

import os

import importlib.util

from PyInstaller.utils.hooks import collect_data_files

block_cipher = None


def rocm_runtime_files():
    """Runtime files of AMD's ROCm PyTorch build (Windows), if it is installed.

    torch loads the ROCm DLLs from these packages by file path at import time, so they
    must stay on disk in their original layout. Compiler tools, import libraries and
    headers are left out.
    """
    files = []
    for package in ('_rocm_sdk_core', '_rocm_sdk_libraries_custom'):
        spec = importlib.util.find_spec(package)
        if spec is None:
            continue
        root = os.path.dirname(spec.origin)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ('__pycache__', 'include')]
            dest = os.path.join(package, os.path.relpath(dirpath, root))
            files += [(os.path.join(dirpath, name), dest) for name in filenames
                      if not name.endswith(('.exe', '.lib', '.h'))]
    return files


rocm_files = rocm_runtime_files()

a = Analysis(
    ['spotless_film_modern.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('weights/*.pth', 'weights'),  # Include model weights
        ('*.py', '.'),  # Include all Python modules
    ] + ([('weights/big-lama.pt', 'weights')]  # Optional LaMa inpainting weights
         if os.path.exists(os.path.join(SPECPATH, 'weights', 'big-lama.pt')) else [])
      + collect_data_files('tkinterdnd2')  # tkdnd native libraries
      + rocm_files,
    hiddenimports=[
        'torch',
        'torchvision', 
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'customtkinter',
        'tkinterdnd2',
        'tkinter',
        'numpy',
        'cv2',
        'threading',
        'dataclasses',
        'enum',
        'typing',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'jupyter',
        'notebook',
        'ipython',
        'pandas',
        'scipy',
        'sklearn',
        'tensorflow',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# With ROCm the bundle is several GB: too large for a single-file exe (4GB limit, and
# it would be unpacked on every start), so build a folder with the exe inside instead
onedir = bool(rocm_files)

exe = EXE(
    pyz,
    a.scripts,
    *([] if onedir else [a.binaries, a.zipfiles, a.datas]),
    [],
    exclude_binaries=onedir,
    name='SpotlessFilm',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True if you want console for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one: 'icon.ico'
)

if onedir:
    coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=False, name='SpotlessFilm')

# For macOS, create an app bundle
app = BUNDLE(
    exe,
    name='SpotlessFilm.app',
    icon=None,  # Add icon path here if you have one: 'icon.icns'
    bundle_identifier='com.spotlessfilm.app',
    info_plist={
        'NSHighResolutionCapable': 'True',
        'NSRequiresAquaSystemAppearance': 'False',
    },
)