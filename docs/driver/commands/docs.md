### docs

    birch docs

Build the reference documentation for the project. This creates a Markdown
file `DOCS.md` in the current working directory.

!!! note
    The Birch documentation system is inspired by JavaDoc and Doxygen. It is suggested to use it similarly.

It will be overwritten if it already exists, and may be readily converted to other formats using a utility such as `pandoc`.

The content of `DOCS.md` is gathered from documentation comments that precede declarations:

    /**
     * Documentation comment.
     */
     class A {
       // ...
     }

    /**
     * Documentation comment.
     */
    function f(a:A, b:B) {
      // ...
    }

    /**
     * Documentation comment.
     */
    a:A;

While the content of these documentation comments is not prescribed, the format should be Markdown, as they are copied verbatim into the `DOCS.md` file where required. It is suggested that the first sentence of each comment is a brief, standalone description, and that parameters are documented using a bulleted list as follows:

    /**
     * Do something.
     *
     * - a: The first parameter.
     * - b: The second parameter.
     */
    function f(a:A, b:B) {
      // ...
    }
