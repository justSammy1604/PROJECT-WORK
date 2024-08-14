// Function to log the click details on the console
function logClick(event) {

    // const tagName = event.target.tagName.toLowerCase();
    // const excludedTags = ['input', 'textarea', 'select', 'form', 'label'];

    // if (excludedTags.includes(tagName)) {
    //     return;
    // }

    const clickDetails = {
        timestamp: new Date().toISOString(),
        element: event.target.tagName,
        textContent: event.target.innerText,
        title: document.title
    };

    console.log("User Clicked:", clickDetails);

    // You can send this data to a server for storage/processing
    // For example, using fetch or XMLHttpRequest

    fetch('http://127.0.0.1:5000/log_click', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(clickDetails)
    }).then(response => response.json())
        .then(data => console.log('Server Respose: ', data))
        .then(error => console.log('Error: ', error));
}

// function logFormSubmission(event) {
//     event.preventDefault(); // Prevent the default form submission behavior

//     const formData = new FormData(event.target); // Get form data
//     const formDetails = {
//         timestamp: new Date().toISOString(),
//         element: 'FORM',
//         formData: {}
//     };

//     // Iterate over the form data and add it to the formDetails object
//     formData.forEach((value, key) => {
//         formDetails.formData[key] = value;
//     });

//     console.log("Form Submitted:", formDetails);

//     fetch('http://127.0.0.1:5000/log_click', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json'
//         },
//         body: JSON.stringify(formDetails)
//     })
//         .then(response => response.json())
//         .then(data => console.log('Server Response:', data))
//         .catch(error => console.log('Error:', error));

//     // Optionally, you can submit the form after logging the data
//     // event.target.submit();
// }

// Add event listener to track clicks on the whole document
document.addEventListener('click', logClick);

// Add event listener to log form submissions
// document.querySelectorAll('form').forEach(form => {
//     form.addEventListener('submit', logFormSubmission);
// });




//CODE FOR LOGGING ONLY BUTTON CLICKS
// function logClick(event) {
//     let clickDetails = null;

//     // Check if the clicked element is a button
//     if (event.target.tagName.toLowerCase() === 'button') {
//         // Populate the clickDetails object
//         clickDetails = {
//             timestamp: new Date().toISOString(),
//             element: event.target.tagName,
//             textContent: event.target.innerText,
//             title: document.title
//     };

//         // Log the clickDetails object to check its content
//         console.log("User Clicked:", clickDetails);
//     }

//     // Only log and send clickDetails if it is not null (i.e., a button was clicked)
//     if (clickDetails) {

//         // Send click data to the server
//         fetch('http://127.0.0.1:5000/log_click', {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json'
//             },
//             body: JSON.stringify(clickDetails,)
//         })
//             .then(response => response.json())
//             .then(data => console.log('Server Response:', data))
//             .catch(error => console.error('Error:', error));
//     }
// }

// // Add event listener to track clicks on the whole document
// document.addEventListener('click', logClick);