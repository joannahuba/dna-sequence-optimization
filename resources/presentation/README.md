
## Uruchomienie

Żeby uruchomic marp'a trzeba wrzucić

```shell
marp other/presentations/nazwa_projektu --pdf --allow-local-files
```

## Dodanie rozszerzenie do VS code
Żeby móć korzystac efektywnie z marp'a trzeba dodać rozszerzeni: `Marp for VS Code
`

## Instalacja

W przypadku WSL chromium trzeba je zainstalować:

```shell
sudo apt update
sudo apt install npm
# mac
brew install node
```

```shell
sudo npm install -g @marp-team/marp-cli
# mac
sudo npm install -g @marp-team/marp-cli
```

```shell
# biblioteki graficzne 
sudo apt install -y libnss3 libxss1 libasound2 libgbm-dev libatk-bridge2.0-0 libgtk-3-0
```

```shell
# instalacja chromium
sudo apt install -y chromium-browser
# mac
brew install --cask chromium
```

marp other/presentations/MBS_project_summary_new.md --pdf --allow-local-files