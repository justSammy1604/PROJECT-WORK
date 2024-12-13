# Set the working directory
#Copy the package.json and package-lock.json files
WORKDIR /app 
#Install the dependencies
COPY package*.json ./  
#Copy the app files
RUN npm install  
#Build the app
COPY . .  
#Expose the port
RUN npm run build  
#Run the app
EXPOSE 3000

CMD ["npm", "start"]
