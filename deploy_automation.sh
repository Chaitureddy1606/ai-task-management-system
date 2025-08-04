#!/bin/bash

# AI Task Management System - Auto Deployment Script
# This script automatically commits changes to Git and triggers Streamlit Cloud deployment

echo "🚀 Starting Auto Deployment Process..."

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "📍 Current branch: $CURRENT_BRANCH"

# Check for changes
if [ -z "$(git status --porcelain)" ]; then
    echo "✅ No changes to commit"
    exit 0
fi

# Add all changes
echo "📁 Adding all changes to git..."
git add .

# Get list of changed files
CHANGED_FILES=$(git diff --cached --name-only)
echo "📝 Changed files:"
echo "$CHANGED_FILES"

# Create commit message with timestamp
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
COMMIT_MESSAGE="Auto update: $TIMESTAMP - $(echo "$CHANGED_FILES" | head -n 3 | tr '\n' ', ' | sed 's/,$//')"

echo "💾 Committing changes..."
git commit -m "$COMMIT_MESSAGE"

# Push to remote repository
echo "📤 Pushing to remote repository..."
git push origin $CURRENT_BRANCH

echo "✅ Changes pushed to Git successfully!"

# Check if Streamlit Cloud is configured
if [ -f ".streamlit/config.toml" ]; then
    echo "🌐 Streamlit Cloud configuration detected"
    echo "📋 Your app will automatically deploy on Streamlit Cloud"
    echo "🔗 Check your Streamlit Cloud dashboard for deployment status"
else
    echo "⚠️  Streamlit Cloud configuration not found"
    echo "📝 Please configure Streamlit Cloud manually"
fi

echo "🎉 Auto deployment process completed!"
echo "⏱️  Deployment will be available in 2-5 minutes" 