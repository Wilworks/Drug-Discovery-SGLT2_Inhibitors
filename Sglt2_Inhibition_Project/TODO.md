# TODO: Fix Git Issues with Virtual Environment and Push to GitHub

- [x] Navigate to Sglt2_Inhibition_Project directory
- [x] Check if 'Wil/' is tracked in git using git ls-files | grep Wil/ (no output, not tracked)
- [x] If tracked, remove 'Wil/' from git tracking using git rm --cached -r Wil/ (skipped, not tracked)
- [x] Set git config core.autocrlf false to prevent line ending issues
- [x] Run git add . to stage changes
- [x] Verify with git status that venv is ignored (Wil/ untracked, ignored)
- [ ] Check git remote -v
- [ ] Set git remote to https://github.com/Wilworks/Drug-Discovery-SGLT2_Inhibitors.git if not set
- [ ] Commit changes with message "Fix venv tracking and update .gitignore"
- [ ] Push to GitHub using git push -u origin main
- [ ] Update this TODO.md to mark all tasks as completed
