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
    html += '<tr><td>' + array[i].faculty_name + '</td><td>'+ array[i].title + '</td><td>' + array[i].research_areas + '</td><td>' + array[i].office  + '</td><td>' + array[i].phone  + '</td><td>' + array[i].email + '</td><td>' + array[i].page + '</td><td>' + array[i].google_scholar_link + '</td></tr>';
  }
  $('#results-table-body').html(html);
}

// button on-click function
let predict = function(){
        let X = get_input_text();
        send_text_json(X);
    };
