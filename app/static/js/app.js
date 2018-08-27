let send_text_json = function() {
    $.ajax({
        url: '/submit',
        contentType: "application/json; charset=utf-8",
        type: 'POST',
        data: JSON.stringify(),
        success: function (data) {
            drawTable(data);
        }
    });
};

let predict = function(){
        send_text_json();
    };

function drawTable(data) {
  var html = '';
  for (var i = 0; i < data.length; i++) {
    html += '<tr><td>' + data[i].object_id + '</td><td>'+ data[i].name + '</td><td>' + data[i].currency + '</td><td>' + data[i].risk_level + '</td><td>' + data[i].acct_category_pred  +'</td><td>' + data[i].predict_fraud  + '</td></tr>';
  }
  $('#results-table-body').html(html);
}
