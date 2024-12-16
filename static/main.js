// static/scripts/main.js

function toggleInputs() {
    const category = document.getElementById("category").value;
    const familyInputs = document.getElementById("family-inputs");
    const otherInputs = document.getElementById("other-inputs");

    if (category === "family") {
        familyInputs.classList.remove("hidden");
        otherInputs.classList.add("hidden");
    } else if (category === "personal") {
        familyInputs.classList.add("hidden");
        otherInputs.classList.remove("hidden");
    } else {
        familyInputs.classList.add("hidden");
        otherInputs.classList.add("hidden");
    }
}
