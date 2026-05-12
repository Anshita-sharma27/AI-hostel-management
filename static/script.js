function startScan(){
    fetch("/scan_face")
    .then(res => res.json())
    .then(data => {

        if(data.name === "Unknown"){
            alert("❌ Face Not Recognized");
        }
        else{
            alert("✔ Attendance Marked: " + data.name);
        }
    });
}

function showToast(msg){
    let t = document.getElementById("toast");
    t.innerHTML = msg;
    t.style.display = "block";

    setTimeout(() => {
        t.style.display = "none";
    }, 3000);
}