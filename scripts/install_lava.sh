#!/bin/bash
# Set the repository URL, branch, and the directory into which to clone/pull the repository.
REPO_URL="https://github.com/lava-nc/lava.git"
BRANCH_NAME="f2f_conv"  # Update this with the branch you want to switch to
BASE_DIR="$HOME"
REPO_DIR="${BASE_DIR}/lava_f2f_conv"

echo "Cloning repository..."
git clone "$REPO_URL" "$REPO_DIR" || exit 1
cd "$REPO_DIR" || exit 1

echo "Switching to branch $BRANCH_NAME..."
git checkout "$BRANCH_NAME" || exit 1

# Install dependencies using pip3
echo "Installing dependencies..."
pip3 install -e .