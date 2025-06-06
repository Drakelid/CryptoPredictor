FROM node:18-alpine

# Use the backend service URL when building inside docker-compose
ARG VITE_API_BASE_URL=http://backend:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install --legacy-peer-deps --force

# Copy application code
COPY . .

# Build the application
RUN npm run build --legacy-peer-deps --force

# Install serve to run the application
RUN npm install -g serve

# Expose port
EXPOSE 3000

# Run the application
CMD ["serve", "-s", "dist", "-l", "3000"]
