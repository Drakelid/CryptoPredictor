FROM node:18-alpine as build

# Set environment variables for build stage
ARG VITE_API_BASE_URL=http://backend:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

WORKDIR /app

# Copy package.json and package-lock.json first for better caching
COPY package*.json ./

# Install dependencies
RUN npm install --legacy-peer-deps --force

# Copy .env file and all application code
COPY .env* ./
COPY . .

# Make sure environment variables are correctly integrated into the build
RUN echo "VITE_API_BASE_URL=$VITE_API_BASE_URL" > .env.production.local

# Build the application
RUN npm run build --legacy-peer-deps --force

# Use a separate stage for the runtime
FROM node:18-alpine as runtime

WORKDIR /app

# Install serve for static file serving and gettext-base for envsubst
RUN npm install -g serve && \
    apk add --no-cache gettext

# Copy built files from the build stage
COPY --from=build /app/dist /app/dist
# Copy the public directory for env-config.js template
COPY --from=build /app/public /app/public
# Copy the environment file for potential runtime use
COPY --from=build /app/.env* /app/

# Copy and make entrypoint script executable
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Expose port
EXPOSE 3000

# Set runtime environment variable (will be used by serve-env-handler)
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

# Run the application with entrypoint script for environment variable substitution
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["serve", "-s", "dist", "-l", "3000"]
