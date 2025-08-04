#!/bin/bash

# Quick Deploy Script for AI Task Management System
# This script quickly commits and pushes changes to trigger Streamlit Cloud deployment

echo "🚀 Quick Deploy - AI Task Management System"

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed"
    exit 1
fi

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Not in a git repository"
    exit 1
fi

# Get current status
echo "📊 Current Git Status:"
git status --short

# Check if there are changes to commit
if [ -z "$(git status --porcelain)" ]; then
    echo "✅ No changes to commit"
    exit 0
fi

# Add all changes
echo "📁 Adding all changes..."
git add .

# Create commit message
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
COMMIT_MSG="Update: $TIMESTAMP - Auto deployment"

echo "💾 Committing changes..."
git commit -m "$COMMIT_MSG"

# Push to remote
echo "📤 Pushing to remote repository..."
git push origin main

echo "✅ Successfully deployed!"
echo "🌐 Streamlit Cloud will automatically deploy in 2-5 minutes"
echo "📋 Check your Streamlit Cloud dashboard for status" 