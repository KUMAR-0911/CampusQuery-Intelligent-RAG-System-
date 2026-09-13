# Build stage
FROM node:20-alpine AS build

WORKDIR /app

# Copy package.json and package-lock.json to leverage Docker cache
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the frontend application code
COPY . .

# Build the Vite React app for production
RUN npm run build

# Production stage using Nginx
FROM nginx:alpine

# Copy the custom Nginx config for React Router
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the built assets from the build stage to Nginx
COPY --from=build /app/dist /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
