function removeRepeats(str) {
	let tempStr = "";
	tempStr = str;
	for(i=15; i>=3; i--) {
		let regexString = "";

		regexString += "(\\w+|[.,]) ".repeat(i);
		
		regexString += "(?:";
		outputString = "";

		for(j=1; j <= i; j++) {
			outputString += `$${j} `;
			if(j == i) {
				regexString += `\\${j}`;
			}else {
				regexString += `\\${j} `;
			}	
		}
		regexString += "\\s?)+";
		regex = new RegExp(regexString, "g")
		console.log(regexString)
		tempStr = tempStr.replace(regex, outputString);
	}
	return tempStr
}

async function simplifyText() {
    let idx = 0;
    let originalContent = [];
    let content = [];

    Array.from(document.getElementsByTagName("p")).forEach((div)=>{
        // do stuff with div.id, div.textContent
        let paragraphContent = div.innerText.replace(/\[.+\]/, "").replace(/[,.!\(\)]/, "");
        paragraphContent = paragraphContent.trim();

        if(paragraphContent != "") {
            let url = "http://ec2-100-26-240-251.compute-1.amazonaws.com:7777/?text=" + encodeURIComponent(paragraphContent);

            let xhr = new XMLHttpRequest();
            xhr.open("GET", url, true);

            let currentIdx = idx;
            idx++;

            content[currentIdx] = "";
            originalContent[currentIdx] = paragraphContent;

            xhr.onload = function() {
                let currentURL = window.location.href;

                content[currentIdx] = removeRepeats(JSON.parse(xhr.responseText)["output"]);
                document.body.innerHTML = `<h1>Simplified Page: ${currentURL}</h1><br><br>`;

                content.forEach(function(value, idx) {
                    document.body.innerHTML += `<div style="background-color: rgb(150, 150, 150); margin: auto; text-align: center"><p><span style='color: red>'>Original : ${originalContent[idx].split("january ").join("")}</span></p></div>`
                    document.body.innerHTML += "<br>"
                    document.body.innerHTML += `<div style="background-color: rgb(200, 200, 200); margin: auto; text-align: center"><p><span style='color: blue>'>Simplified : ${value.split("january ").join("")}</span></p></div>`
                    document.body.innerHTML += "<br><br>"
                })
            }
            xhr.send();
        }
    });
}


