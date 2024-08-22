//Function to log the clicks of the user
function logClick(event) {

    const tagName = event.target.tagName.toLowerCase();
    const excludedTags = ['input', 'textarea', 'select', 'form', 'label'];

    if (excludedTags.includes(tagName)) {
        return;
    }

    //Elements that are stored in the JSON file 
    const clickDetails = {
        timestamp: new Date().toISOString(), //timestamp of when the user interacted with the element
        element: event.target.tagName, //the type of element the user clicked (buttons, a tags, forms etc.)
        textContent: event.target.innerText, //the text that the button holds (Student management system, Training and placement system etc.)
        title: document.title //title of the web page
    };

    console.log("User Clicked:", clickDetails);

    fetch('http://127.0.0.1:5000/log_click', {  //port number of where the flask code is running
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(clickDetails)
    }).then(response => response.json())
        .then(data => console.log('Server Respose: ', data))
        .then(error => console.log('Error: ', error));
}

//function for logging the users intercation with forms
function logFormSubmission(event) {
    event.preventDefault(); // Prevent the default form submission behavior

    const formData = new FormData(event.target); // Get form data
    const formDetails = {
        timestamp: new Date().toISOString(),
        element: 'FORM',
        title: document.title,
        formData: {}
    };

    // Iterate over the form data and add it to the formDetails object
    formData.forEach((value, key) => {
        formDetails.formData[key] = value;
    });

    console.log("Form Submitted:", formDetails);

    fetch('http://127.0.0.1:5000/log_click', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formDetails)
    })
        .then(response => response.json())
        .then(data => console.log('Server Response:', data))
        .catch(error => console.log('Error:', error));

    // Optionally, you can submit the form after logging the data
    //event.target.submit();
}

// Add event listener to track clicks on the whole document
document.addEventListener('click', logClick);

// Add event listener to log form submissions
document.querySelectorAll('form').forEach(form => {
    form.addEventListener('submit', logFormSubmission);
});
