const searchWrapper = document.querySelector(".search-box");
const inputBox = searchWrapper.querySelector("input");
const suggBox = searchWrapper.querySelector(".autoSugg");
const icon = searchWrapper.querySelector(".icon");


if(searchWrapper===null){
	console.log("1");
}
if(inputBox==null){
	console.log("2");
}
/*When User Type Any Key and Release*/
else{
inputBox.onkeyup=(e)=>{
	let userData = e.target.value; /*User Entered Data Stored in userData*/
	console.log(5+6);
	let emptyAr = [];
	if(userData){

		emptyAr = suggestions.filter((data)=>{
				/*Convert input and data characters to lowercase to avoid case-sensitivity conflicts*/
				return data.toLocaleLowerCase().startsWith(userData.toLocaleLowerCase());
		});

		emptyAr = emptyAr.map((data)=>{
			/*returned data from above funtion is passed into list tag*/
			return data = '<li>'+data+'<li>';
		});
		/* To Show AutoComplete Box */
		searchWrapper.classList.add("active");
		getSuggestions(emptyAr);
		let allList = suggBox.querySelectorAll("li");
		for (let i = 0; i < allList.length; i++) {
				/* When Clicked from the dropdown, copy it into search box*/
				allList[i].setAttribute("onclick","select(this)");
		}
	}else{
		searchWrapper.classList.remove("active");/*Hide AutoComplete Box*/
	}
}
function select(element){
	let selectData = element.textContent;
	inputBox.value = selectData;
	searchWrapper.classList.remove("active");
}
function getSuggestions(list){
	let listData;
	if(!list.length){
		userValue = inputBox.value;
		listData ='<li>'+userValue+'<li>';
	}else{
		listData = list.join('');
	}
	suggBox.innerHTML = listData;
}
}