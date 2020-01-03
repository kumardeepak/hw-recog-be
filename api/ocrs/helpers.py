import json

def get_table_structure():
    rsp = {
            "status": {
                "code": 200,
                "message": "request successful"
            },
            "response": [
                {
                    "header": {
                        "row": 10,
                        "col": 5,
                        "title": "Marks tally"
                    },
                    "data": [
                        {"row": 0, "col": 0, "text": "S.NO"},
                        {"row": 0, "col": 1, "text": "Max Mark"},
                        {"row": 0, "col": 2, "text": "Allowed Mark"},
                        {"row": 0, "col": 3, "text": "Total Marks"},
                        {"row": 0, "col": 4, "text": "Result"},

                        {"row": 1, "col": 0, "text": "1"},
                        {"row": 1, "col": 1, "text": "2"},
                        {"row": 1, "col": 2, "text": "3"},
                        {"row": 1, "col": 3, "text": "4"},
                        {"row": 1, "col": 4, "text": "5"},

                        {"row": 3, "col": 0, "text": ""},
                        {"row": 3, "col": 1, "text": "3"},
                        {"row": 3, "col": 2, "text": "0"},
                        {"row": 3, "col": 3, "text": "1"},
                        {"row": 3, "col": 4, "text": "3"},

                        {"row": 4, "col": 0, "text": "4"},
                        {"row": 4, "col": 1, "text": "5"},
                        {"row": 4, "col": 2, "text": "6"},
                        {"row": 4, "col": 3, "text": ""},
                        {"row": 4, "col": 4, "text": "3"},

                        {"row": 5, "col": 0, "text": "3"},
                        {"row": 5, "col": 1, "text": "4"},
                        {"row": 5, "col": 2, "text": "6"},
                        {"row": 5, "col": 3, "text": "7"},
                        {"row": 5, "col": 4, "text": "1"},

                        {"row": 6, "col": 0, "text": "3"},
                        {"row": 6, "col": 1, "text": "6"},
                        {"row": 6, "col": 2, "text": "3"},
                        {"row": 6, "col": 3, "text": "3"},
                        {"row": 6, "col": 4, "text": "3"},

                        {"row": 7, "col": 0, "text": "4"},
                        {"row": 7, "col": 1, "text": "3"},
                        {"row": 7, "col": 2, "text": "3"},
                        {"row": 7, "col": 3, "text": "10"},
                        {"row": 7, "col": 4, "text": "2"},

                        {"row": 8, "col": 0, "text": "10"},
                        {"row": 8, "col": 1, "text": "12"},
                        {"row": 8, "col": 2, "text": "2"},
                        {"row": 8, "col": 3, "text": "3"},
                        {"row": 8, "col": 4, "text": "4"},

                        {"row": 9, "col": 0, "text": "5"},
                        {"row": 9, "col": 1, "text": "3"},
                        {"row": 9, "col": 2, "text": "2"},
                        {"row": 9, "col": 3, "text": "2"},
                        {"row": 9, "col": 4, "text": "1"},
                        ]
                },

                {
                    "header": {
                        "row": 7,
                        "col": 2,
                        "title": "Student information"
                    },
                    "data": [
                        {"row": 0, "col": 0, "text": "2"},
                        {"row": 0, "col": 1, "text": "2"},

                        {"row": 1, "col": 0, "text": "3"},
                        {"row": 1, "col": 1, "text": "3"},

                        {"row": 2, "col": 0, "text": "4"},
                        {"row": 2, "col": 1, "text": "5"},

                        {"row": 3, "col": 0, "text": "4"},
                        {"row": 3, "col": 1, "text": "2"},

                        {"row": 4, "col": 0, "text": "4"},
                        {"row": 4, "col": 1, "text": "2"},

                        {"row": 5, "col": 0, "text": "2"},
                        {"row": 5, "col": 1, "text": "6"},

                        {"row": 6, "col": 0, "text": "2"},
                        {"row": 6, "col": 1, "text": "6"},
                    ]
                },
            ]
        }

    return rsp