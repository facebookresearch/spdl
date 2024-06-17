/**
 * We add extra br tags to the autodoc output, so each parameter is shown on
 * its own line.
 */
function setupAutodocPy() {
    const paramElements = document.querySelectorAll('.py .sig-param')

    Array(...paramElements).forEach((element) => {
        let brElement = document.createElement('br')
        element.parentNode.insertBefore(brElement, element)
    })

    const lastParamElements = document.querySelectorAll('.py em.sig-param:last-of-type')

    Array(...lastParamElements).forEach((element) => {
        let brElement = document.createElement('br')
        element.after(brElement)
    })
}

function setupAutodocCpp() {
    const highlightableElements = document.querySelectorAll(".c dt.sig-object, .cpp dt.sig-object")

    Array(...highlightableElements).forEach((element) => {
        element.classList.add("highlight");
    })

    const documentables = document.querySelectorAll("dt.sig-object.c,dt.sig-object.cpp");

    Array(...documentables).forEach((element) => {
        element.classList.add("highlight");

        var parens = element.querySelectorAll(".sig-paren");
        var commas = Array(...element.childNodes).filter(e => e.textContent == ", ")

        if (parens.length != 2) return;

        commas.forEach(c => {
            if (c.compareDocumentPosition(parens[0]) == Node.DOCUMENT_POSITION_PRECEDING &&
                c.compareDocumentPosition(parens[1]) == Node.DOCUMENT_POSITION_FOLLOWING
            ) {
                let brElement = document.createElement('br')
                let spanElement = document.createElement('span')
                spanElement.className = "sig-indent"
                c.after(brElement)
                brElement.after(spanElement)
            }
        });

        if (parens[0].nextSibling != parens[1]) {
            // not an empty argument list
            let brElement = document.createElement('br')
            let spanElement = document.createElement('span')
            spanElement.className = "sig-indent"
            parens[0].after(brElement)
            brElement.after(spanElement)
            let brElement1 = document.createElement('br')
            parens[1].parentNode.insertBefore(brElement1, parens[1]);
        }
    })
}

document.addEventListener("DOMContentLoaded", function() {
    setupAutodocPy()
    setupAutodocCpp()
})
