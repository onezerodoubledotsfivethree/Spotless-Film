# Building SpotlessFilm for Windows

## For Your Friend to Build on Windows:

### Prerequisites
1. **Download and install Python 3.9+** from https://python.org
   - ✅ Check "Add Python to PATH" during installation

### Build Steps
1. **Download the source code** (get the entire project folder)

2. **Open Command Prompt** (cmd) or PowerShell

3. **Navigate to the src folder:**
   ```cmd
   cd path\to\Dust-Removal-UNet\src
   ```

4. **Download the model weights:**
   - Download the archive from Google Drive: https://drive.google.com/file/d/1yR4gk2SgU0-p_EOrihckt3gFE8_lYyq7/view?usp=sharing
   - Create a `weights` folder inside `src` (if it doesn't exist yet)
   - Extract the archive so that the `.pth` files end up **directly** in `src\weights\`:
     ```
     Dust-Removal-UNet\
     └── src\
         ├── spotless_film_modern.py
         └── weights\
             └── v5_bce_unet_epoch30.pth
     ```
   - ⚠️ Make sure there is no extra nested folder (e.g. `src\weights\weights\...`) —
     the `.pth` files must be right inside `src\weights\`, otherwise the build fails
     or the app can't find the model.

   - **Optional, for LaMa inpainting:** download
     [big-lama.pt](https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt) (~200MB)
     into the same `src\weights\` folder. This enables the OpenCV/LaMa switch in the
     Dust Removal section; without the file the app uses OpenCV only.

5. **Install required packages:**
   ```cmd
   pip install pyinstaller torch torchvision pillow customtkinter tkinterdnd2 opencv-python numpy
   ```

6. **Build the executable:**
   ```cmd
   python build_executable.py
   ```

7. **Find the .exe file:**
   - Look in `distribution\SpotlessFilm.exe`
   - This is the Windows executable!

## Alternative Simple Build
If the automated script doesn't work (the weights from step 4 must already be in `src\weights\`):

```cmd
cd src
pip install pyinstaller customtkinter tkinterdnd2
pyinstaller --onefile --windowed --collect-all tkinterdnd2 --add-data "weights\*.pth;weights" --name SpotlessFilm spotless_film_modern.py
```

The .exe will be in `dist\SpotlessFilm.exe`

To bundle LaMa in the simple build, add `--add-data "weights\big-lama.pt;weights"`.

## File Size
Expect ~300-800MB for the Windows executable (includes all dependencies).

## Sharing
Once built, just send the `SpotlessFilm.exe` file - no installation needed!