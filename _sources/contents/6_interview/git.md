
# Git Essentials 🌟

## 1. What is Git? 📦

**Git** is a distributed version control system that:
- Tracks code changes with precision
- Enables team collaboration
- Safeguards against data loss
- Supports non-linear development via branching

---

## Key Features of Git

1. **Version Control**: Tracks changes in files and enables reverting to previous versions.
2. **Distributed System**: Every developer has a complete copy of the project repository, ensuring reliability.
3. **Branching and Merging**: Allows creating separate branches for features or fixes and merging them into the main project without conflicts.
4. **Collaboration**: Facilitates teamwork by allowing multiple developers to work on the same project seamlessly.
5. **Efficiency**: Fast and lightweight, even for large projects.

---

## Core Concepts

### 1. Repository (Repo)
A storage location where all your code and history are saved.
- **Local Repository**: Stored on your computer.
- **Remote Repository**: Stored on a remote server (e.g., GitHub, GitLab).

### 2. Staging Area
A temporary space to prepare changes before committing.

### 3. Commit
A snapshot of your code at a specific time, with a message describing the changes.

### 4. Branch
A separate line of development (e.g., `master`, `feature-branch`).

### 5. Merge
Combines changes from one branch into another.

### 6. Pull
Fetches and integrates changes from a remote repository.

### 7. Push
Sends local changes to a remote repository.

---

## Why Use Git?

1. Keeps a detailed history of every change.
2. Makes collaboration easier by managing code conflicts.
3. Helps in experimenting safely by using branches.
4. Works offline for local development.

Git is a must-have tool for any developer, enabling efficient, organized, and collaborative software development!

---

## Git Commands

### File and Directory Operations
```sh
ls                  # Lists files in the current directory
ls -a               # Lists all files, including hidden files (e.g., .git)
rm -rf .git         # Removes the .git folder to uninitialize a Git repository
cd ..               # Moves one directory up
clear               # Clears the terminal screen
```

### Git Basics
```sh
git init            # Initializes a new Git repository
git status          # Shows the current status of the repository
git add .           # Stages all changes in the directory for commit
git commit -m "message" # Commits staged changes with a descriptive message
```

### Configuration
```sh
git config --global user.email "your_email"    # Sets the global email for commits
git config --global user.name "your_name"      # Sets the global username for commits
git config --list                              # Lists all current Git configuration settings
```

### Branch Management
```sh
git branch                    # Lists all branches in the repository
git branch <branch_name>       # Creates a new branch
git branch -d <branch_name>    # Deletes a branch
git checkout <branch_name>     # Switches to a specific branch
git merge <branch_name>        # Merges the specified branch into the current branch
```

### Viewing and Logging
```sh
git log               # Shows the commit history
git show <commit_hash> # Displays the details of a specific commit
```
*Press `q` to exit log*

### Restoring and Resetting
```sh
git restore <file>    # Restores a specific file to the last committed state
git restore .         # Restores all files to the last committed state
git rm --cached <file> # Removes a file from the staging area (keeps it in the working directory)
git reset .           # Unstages all staged files
```

### Chaining Commands
```sh
git add file && git commit -m "message"  # Stages and commits changes in a single command
git add file & git commit -m "message"   # Runs commands independently
```

### Stashing
```sh
git stash                # Temporarily saves changes without committing
git stash list           # Lists all stashed changes
git stash apply stash@{n} # Applies a specific stash (n is the stash index)
git stash clear          # Clears all stashes
```

### Push and Pull
```sh
git push origin <branch_name>    # Pushes the branch to the remote repository
git push --all origin            # Pushes all branches
git push origin --tags           # Pushes tags
git push --force                 # Overwrites remote changes with your local branch (use with caution)
```

---

## Extra Commands
```sh
history      # Displays the command history
clear        # Clears the terminal screen
git --version # Displays the installed Git version
which git    # Shows the path where Git is installed
```

---

## Example Workflow
```sh
git init
git add .
git commit -m "Initial commit"
git branch feature-branch
git checkout feature-branch
git add file.txt
git commit -m "Added file.txt"
git checkout master
git merge feature-branch
git log
```

---
