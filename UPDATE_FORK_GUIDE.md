# 🔄 How to Update Your Forked Repository

This guide explains how to keep your forked FPL-Elo-Insights repository synchronized with the original repository.

## 📋 One-Time Setup

### 1. Add the Upstream Remote
```bash
# Add the original repository as upstream (replace ORIGINAL_OWNER with actual username)
git remote add upstream https://github.com/ORIGINAL_OWNER/FPL-Elo-Insights.git

# Verify your remotes
git remote -v
```

You should see:
```
origin    https://github.com/firmanhadi21/FPL-Elo-Insights.git (fetch)
origin    https://github.com/firmanhadi21/FPL-Elo-Insights.git (push)
upstream  https://github.com/ORIGINAL_OWNER/FPL-Elo-Insights.git (fetch)
upstream  https://github.com/ORIGINAL_OWNER/FPL-Elo-Insights.git (push)
```

## 🔄 Regular Update Process

### 2. Fetch Latest Changes
```bash
# Fetch the latest changes from upstream
git fetch upstream

# Switch to your main branch
git checkout main

# Merge upstream changes into your main branch
git merge upstream/main
```

### 3. Push Updates to Your Fork
```bash
# Push the updated main branch to your fork
git push origin main
```

## 🚀 Alternative: Using GitHub CLI (if installed)
```bash
# Sync your fork with one command
gh repo sync firmanhadi21/FPL-Elo-Insights
```

## 🔧 Advanced Scenarios

### If You Have Local Changes
```bash
# Stash your changes before updating
git stash

# Update as above
git fetch upstream
git merge upstream/main
git push origin main

# Restore your changes
git stash pop
```

### If You Have Conflicting Changes
```bash
# Create a backup branch first
git checkout -b backup-branch

# Switch back to main and force update (⚠️ This overwrites local changes)
git checkout main
git reset --hard upstream/main
git push origin main --force
```

### Create a Pull Request from Updates
If you want to contribute your changes back:
```bash
# Create a feature branch
git checkout -b feature/my-improvements

# Make your changes and commit
git add .
git commit -m "Add my improvements"

# Push to your fork
git push origin feature/my-improvements

# Then create a PR on GitHub from your fork to the original repo
```

## 📅 Recommended Workflow

1. **Before starting work**: Always update your fork first
2. **Regular intervals**: Check for updates weekly or before major work
3. **Before submitting PRs**: Ensure your fork is up-to-date

## 🔍 Check for Updates
```bash
# See what's different between your fork and upstream
git fetch upstream
git log HEAD..upstream/main --oneline

# See detailed changes
git diff HEAD upstream/main
```

## ⚠️ Important Notes

- Always commit or stash your local changes before updating
- If you've modified files that were also changed upstream, you may need to resolve merge conflicts
- Consider creating feature branches for your work to avoid conflicts with main branch updates
- Your `origin` points to your fork, `upstream` points to the original repository

## 🆘 Troubleshooting

### "Already up to date" but GitHub shows behind
```bash
git fetch upstream --prune
git merge upstream/main
```

### Reset fork to match upstream exactly
```bash
git fetch upstream
git checkout main
git reset --hard upstream/main
git push origin main --force
```

---

💡 **Pro Tip**: Set up a GitHub Action to automatically sync your fork, or use GitHub's "Sync fork" button in the web interface for simple updates.
