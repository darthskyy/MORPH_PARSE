// getting all the sections and nav links
const sections = document.querySelectorAll('section');
const navLinks = document.querySelectorAll('nav ul li a');

// Detecting the current section and adding active class to the nav link
function activateSection() {
    let currentSection = '';

    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        if (pageYOffset >= sectionTop - 60) {
            currentSection = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href').includes(currentSection)) {
            link.classList.add('active');
        }
    });
}

// Event listener for scrolling
window.addEventListener('scroll', activateSection);

// this is to highlight all the best scores when the column or row is hovered over. Makes the data even more readable and easy to understand
// How it works (for my understanding):
// get all the th elements in the tr element with the class "score-heading" and log out their values (and index in the parent) when element is hovered over
// when that happens highlight the cells in the table of the same index as the th element hovered over with the class "best"
// when the mouse leaves the th element remove the highlight from the cells in the table

const scoreHeadings = document.querySelector('tr.score-heading');
const thElements = scoreHeadings.querySelectorAll('th');
thElements.forEach(th => {
    th.addEventListener('mouseover', () => {
        console.log(th.textContent);
        const index = Array.from(th.parentElement.children).indexOf(th);
        const tableRows = document.querySelectorAll('table tr');
        tableRows.forEach(row => {
            const cells = row.querySelectorAll('td.best');
            cells.forEach((cell, i) => {
                const cellIndex = Array.from(cell.parentElement.children).indexOf(cell);
                if (cellIndex === index) {
                    cell.style.backgroundColor = '#FFF59D';
                }
            });
        });
    });
});

thElements.forEach(th => {
    th.addEventListener('mouseout', () => {
        console.log(th.textContent);
        const index = Array.from(th.parentElement.children).indexOf(th);
        const tableRows = document.querySelectorAll('table tr');
        tableRows.forEach(row => {
            const cells = row.querySelectorAll('td.best');
            cells.forEach((cell, i) => {
                const cellIndex = Array.from(cell.parentElement.children).indexOf(cell);
                if (cellIndex === index) {
                    cell.style.backgroundColor = '';
                    cell.style.color = '';
                }
            });
        });
    });
});




// Function to highlight table row cells with class "best"
function highlightBestCells(event) {
    const row = event.currentTarget;
    const bestCells = row.querySelectorAll('td.best');
    bestCells.forEach(cell => {
        cell.style.backgroundColor = '#FFF59D';
    });
}

// Function to remove highlight from table row cells with class "best"
function removeHighlightBestCells(event) {
    const row = event.currentTarget;
    const bestCells = row.querySelectorAll('td.best');
    bestCells.forEach(cell => {
        cell.style.backgroundColor = '';
    });
}

// Adding event listeners to each table row
const tableRows = document.querySelectorAll('table tr');
tableRows.forEach(row => {
    row.addEventListener('mouseover', highlightBestCells);
    row.addEventListener('mouseout', removeHighlightBestCells);
});