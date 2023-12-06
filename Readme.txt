FIRST:
pip install -r requirements.txt

SECOND:
TO INSTALL DLIB:
1) install Cmake : https://github.com/Kitware/CMake/releases/download/v3.24.2/cmake-3.24.2-windows-x86_64.msi
2) Install Dlib File as per python version ( preferred Python 3.7)
   https://github.com/sachadee/Dlib/blob/main/dlib-19.22.99-cp37-cp37m-win_amd64.whl
3)Copy the path of downloaded dlib file (eg: C:\Users\Siddhesh Gadkar\Downloads\dlib-19.22.99-cp37-cp37m-win_amd64.whl)
4)Open terminal or CMD and type pip install "above path"

5)in app.py make sure all the paths to folders have the right path.(Eg. [app.py line-80] You need to change the directory path of warnings folder name as per your local device.)