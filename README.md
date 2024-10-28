git add . ':(exclude)data/*'
git add -- . ':(exclude)data'

git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/luannt0801/FedCSP.git
git push -u origin main