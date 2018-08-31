let get_input_text = function() {
    let X = $("input#text_input").val()
    return {"text_input": X}
    };

let send_text_json = function(X) {
    $.ajax({
        url: '/submit',
        contentType: "application/json; charset=utf-8",
        type: 'POST',
        success: function (data) {
            drawTable(data);
        },
        data: JSON.stringify(X) //This gets sent to the submit page.
    });
};

function drawTable(data) {
  var html = '';
  let array = JSON.parse(data);
  for (var i = 0; i < array.length; i++) {
    img_name = "post.png"
    if (array[i].university_name == 'Stanford University') {
        img_name = 'stanford.png'
    }
    else if (array[i].university_name == 'Texas A&M University--College Station') {
        img_name = 'tamu.png'
    }
    else if (array[i].university_name == 'University of Texas--Austin (Cockrell)'){
        img_name = 'utaustin.png'
    }
    else if (array[i].university_name == 'University of Tulsa'){
        img_name = 'utulsa.png'
    }

    html += '<div class="single-post d-flex flex-row"><div class="thumb"><img src="img/' + img_name + '" alt=""></div><div class="details" style="padding-left: 20px"><div class="title d-flex flex-row justify-content-between"><div class="titles"><h4>' + array[i].faculty_name + '</h4><h6><i>' + array[i].title + '</i>, ' + array[i].university_name + '</h6></div></div><p>' + array[i].research_areas + '</p></div></div>'
  }
  $('#facecards').html(html);
  $('#facecards')[0].scrollIntoView();
}

// button on-click function
let predict = function(){
        let X = get_input_text();
        send_text_json(X);
    };

    