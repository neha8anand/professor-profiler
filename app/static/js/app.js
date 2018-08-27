let get_input_text = function() {
    let X = $("input#text_input").val()
    return {"text_input": X}
    };

let send_text_json = function(X) {
    $.ajax({
        url: '/submit',
        contentType: "application/json; charset=utf-8",
        type: 'POST',
        data: JSON.stringify(X), //This gets sent to the submit page.
        success: function (data) {
            drawTable(data);
        }
    });
};

function drawTable(data) {
  var html = '';
  for (var i = 0; i < data.length; i++) {
    html += '<tr><td>' + data[i].faculty_name + '</td><td>'+ data[i].faculty_title + '</td><td>' + data[i].research_areas + '</td><td>' + data[i].predicted_research_areas + '</td><td>' + data[i].office  +'</td><td>' + data[i].email + '</td><td>' + data[i].phone + '</td><td>' + data[i].page + '</td><td>' + data[i].google_scholar_link + '</td></tr>';
  }
  $('#results-table-body').html(html);
}

let predict = function(){
        let X = get_input_text();
        send_text_json(X);
    };
