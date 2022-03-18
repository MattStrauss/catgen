# CMPSC 445W Team Project
Using Python, Flask and Tailwind CSS

**Basic Scaffolding from: https://github.com/blackbaba/Flask-Tailwind-Starter**

## Key Features
- Automatically categorizes books based on Cover and Title
- Machine Learning techniques learned in course applied to problem

## How to setup in your local machine
[Setup and activate a virtual python environment](https://flask.palletsprojects.com/en/2.0.x/installation/) before proceeding.

Clone this repository onto your local computer and run:

1. `pip install -r requirements.txt` to install flask packages
2. `npm install` to install npm packages from `package.json`
3. In one terminal run `npm run dev` to run Tailwind in JIT watch mode during development - this will start real time compilation of styles used in your HTML templates
   1. Note: You often will have to hit `shift+cmd+R` to see the changes made and auto-compiled to the CSS, this happens when new Tailwind CSS styles are added
4. In second terminal run `python run.py` to start the Flask development server (debug mode is ON). As you add/remove Tailwind classes in HTML templates, the watcher running in step 3 will automatically regenerate your `app\static\main.css` file which is picked up the flask server running in step 4.
5. When ready for production, kill the Flask development server in step 3 and run  `npm run build:prod` to prepare CSS build ready for production
