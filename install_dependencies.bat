@echo off
echo Installation des dependances Python pour le scraping Emeis...
echo.

pip install selenium beautifulsoup4 pandas openpyxl requests

echo.
echo Dependencies installees!
echo.
echo ATTENTION: Vous devez encore telecharger ChromeDriver:
echo 1. Allez sur https://chromedriver.chromium.org/
echo 2. Telechargez la version correspondant a votre Chrome
echo 3. Ajoutez chromedriver.exe a votre PATH ou placez-le dans le meme dossier que ce script
echo.
pause
