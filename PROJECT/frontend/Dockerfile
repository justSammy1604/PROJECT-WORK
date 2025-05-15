# Use node:20-alpine base image [[5]]
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Install Chromium dependencies (required for Playwright) [[3]][[6]]
#RUN apk add --no-cache chromium firefox
RUN apk add --no-cache \
    chromium \
    firefox \
    nss \
    freetype \
    harfbuzz \
    ca-certificates \
    ttf-freefont \
    nodejs \
    yarn \
    udev \
    bash \
    && rm -rf /var/cache/*

# Copy package.json and install dependencies
COPY package.json ./
RUN npm install

# Explicitly install Playwright browsers (resolves missing executable error) [[5]][[9]]
RUN npx playwright install chromium firefox

# Copy remaining files
COPY . .

# Expose port and run command
EXPOSE 4000
CMD ["npm", "run", "dev"]