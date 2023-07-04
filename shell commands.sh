#initialize git and dvc
git init
dvc init
dvc add data/W2/athetes.csv
dvc remote add -d storage gdrive://<fileid>
git add .
git commit -m "commit message"
dvc push
git push
git tag v1.0
git push --tag

#data v2
dvc add data/W2/athetes.csv
dvc push
git commit data/W2/athletes.csv.dvc -m "Dataset Updates"
git push
git tag v2.0
git push --tag

#switching between versions
git checkout v1.0
dvc checkout
