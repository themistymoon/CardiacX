# --- STAGE 1: Build ---
FROM node:20-alpine as build

# Set the working directory
WORKDIR /app

# Copy package configuration files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application files
COPY . .

# Build the application
RUN npm run build


# --- STAGE 2: Production ---
FROM nginx:stable-alpine as stage-1
# Copy the built assets from the build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy the Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"] 