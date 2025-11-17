#!/bin/bash
# Generate secure secrets for ERCP Protocol

echo "Generating secure secrets..."
echo ""

JWT_SECRET=$(openssl rand -hex 32)
APP_SECRET=$(openssl rand -hex 32)

echo "Add these to your .env file:"
echo ""
echo "JWT_SECRET_KEY=$JWT_SECRET"
echo "APP_SECRET_KEY=$APP_SECRET"
echo ""
echo "⚠️  Keep these secrets secure and never commit them to Git!"
