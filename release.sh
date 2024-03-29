#!/bin/bash

printf "\n#######################################################\n"
echo have you confirmed that directory is ready for release?
echo make clean
echo make docs
echo make coverage
echo update HISTORY.rst
echo git commit


printf "#######################################################\n\n"
echo if you are giving an explicit version have you run bumpversion?
printf "REMOVE 'dev' from version tag (find/replace)\n"
printf "#######################################################\n\n"

printf "\n#######################################################\n"
printf "make sure to create a release on github\n"
printf "https://github.com/stephanwaz/raytools/releases\n"
printf "\n#######################################################\n"

echo -n "proceed to release (y/n)? "
read -r answer

if [ "$answer" != "${answer#[Yy]}" ] ;then
    clean=$(git status --porcelain --untracked-files=no | wc -l)
    if [ "$clean" -lt 1 ]; then
        if [[ $# == 1 && ($1 == "patch" || $1 == "minor" || $1 == "major" || $1 == v*.*.* || $1 == "continue") ]]; then
            if [[ $1 == v*.*.* ]]; then
                git tag -a "$1" -m "tagged for release $1"
            elif [[ $1 == "continue" ]]; then
                echo "using current commit"
            else
                bumpversion --tag --commit "$1"
            fi
            make clean
            python -m build
            echo -n "ok to push (y/n)? "
            read -r answer
            if [ "$answer" != "${answer#[Yy]}" ] ;then
                twine upload dist/*.tar.gz dist/*.whl
                git push
                tag="$(git tag | tail -1)"
                git push origin $tag
			else
				git status
			fi
        else
            echo usage: ./release.sh "[patch/minor/major]"
            echo usage: ./release.sh "vX.X.X (assumes bumpversion has already been run and vX.X.X matches)"
            echo usage: ./release.sh "continue (for picking up an aborted release, run after git commit --amend)"
        fi
    else
        echo working directory is not clean!
        git status --porcelain --untracked-files=no
    fi
else
    echo aborted
fi


