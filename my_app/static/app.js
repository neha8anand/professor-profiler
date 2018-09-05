let get_input_text = function () {
    let X = $("input#text_input").val()
    return { "text_input": X }
};

let send_text_json = function (X) {
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
        research_areas_html = ""
        array[i].research_areas.forEach(research_area => {
            research_areas_html += `<li>` + research_area + `</li>`
        });

        html += `
            <div class="col-lg-8 post-list">
                <div class="single-post d-flex flex-row">
                    <div class="thumb">
                        <img src="` + array[i].photo_link + `" alt="` + array[i].faculty_name + `" width="150" height="200">
                    </div>
                    <div class="details" style="padding-left: 20px">
                        <div class="title d-flex flex-row justify-content-between">
                            <div class="titles">
                                <a href="professor/` + array[i].id + `">
                                    <h3>` + array[i].faculty_name + `</h3>
                                </a>
                                <h5><b>` + array[i].university_name + `</b></h5>
                                <h6><b>Designation:</b> ` + array[i].title + `</h6>
                                <h6><b>Email:</b> ` + array[i].email + `</h6>
                                <h6><b>Phone:</b> ` + array[i].phone + `</h6>
                                <h6><b>Office:</b> ` + array[i].office + `</h6>
                                <h6><b>Website:</b> <a href="` + array[i].page + `">` + array[i].page + `</a></h6>
                            </div>
                        </div>
                    </div>
                </div>
			</div>`
    }

    $('#facecards-section').addClass("section-gap");
    $('#facecards').html(html);
    $('#facecards')[0].scrollIntoView(true, { behavior: 'smooth' });
    window.scrollBy(0, -70);
}

// button on-click function
let predict = function () {
    let X = get_input_text();
    send_text_json(X);
};

// select on-change function
let plot_topic_distribution = function (prof_id) {
    if (prof_id == "-1") {
        $('#plot').html('')
    }
    else {
        $.ajax({
            url: '/professor-topics/' + prof_id,
            contentType: "application/json; charset=utf-8",
            type: 'POST',
            success: function (response) {
                $('#plot').html(response)
            }
        });
    }    
    
};

