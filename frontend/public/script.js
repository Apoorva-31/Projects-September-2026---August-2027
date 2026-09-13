function login() {

    let email = document.getElementById("email").value;
    let password = document.getElementById("password").value;

    if(email === "" || password === ""){
        alert("Please fill all fields!");
        return;
    }

    alert("Login Successful!");
    window.location.href="dashboard1.html";
}